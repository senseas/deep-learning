package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.Objects;
import java.util.Optional;

@Data
public class EnumConstantDeclaration extends Declaration {
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;

    public EnumConstantDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof EnumDeclaration)) return;
        if (node instanceof EnumConstantDeclaration) return;

        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m instanceof Name) {
                    EnumConstantDeclaration declare = new EnumConstantDeclaration(node.getPrarent());
                    declare.setName((Name) m);
                    Optional<Node> first = a.getChildrens().stream().filter(o -> Objects.nonNull(o.getTokenType()) && o.getTokenType().equals(TokenType.COMMA)).findFirst();
                    if (!first.isPresent()) {
                        declare.setChildrens(a.getChildrens());
                        a.getChildrens().remove(m);

                        if (n instanceof ParametersExpression) {
                            declare.setParameters((ParametersExpression) n);
                            declare.getChildrens().add(n);
                            a.getChildrens().remove(n);
                        }

                        if (b instanceof BlockStatement) {
                            declare.setBody((BlockStatement) b);
                            declare.getChildrens().add(b);
                            node.getChildrens().remove(b);
                            list.remove(b);
                        }

                        node.replace(a, declare);
                        c.clear();
                    } else {
                        a.getChildrens().remove(m);
                        if (n instanceof ParametersExpression) {
                            declare.setParameters((ParametersExpression) n);
                            declare.getChildrens().add(n);
                            a.getChildrens().remove(n);
                        }

                        if (b instanceof BlockStatement) {
                            declare.setBody((BlockStatement) b);
                            declare.getChildrens().add(b);
                            node.getChildrens().remove(b);
                            list.remove(b);
                        }

                        int index = node.getChildrens().indexOf(a);
                        node.getChildrens().add(index, declare);
                        c.clear();
                    }
                } else if (m.equals(TokenType.COMMA)) {
                    a.getChildrens().remove(m);
                }
            });
        });
    }

}