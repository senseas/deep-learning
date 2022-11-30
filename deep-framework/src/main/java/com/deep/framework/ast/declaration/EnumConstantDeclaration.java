package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import lombok.Data;

import java.util.Objects;

@Data
public class EnumConstantDeclaration extends Declaration {
    private Name name;
    private ParametersExpression parameters;
    private BlockStatement body;
    private static Node comma;

    public EnumConstantDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (!(node.getPrarent() instanceof EnumDeclaration)) return;
        if (node instanceof EnumConstantDeclaration) return;
        comma = new Node(TokenType.COMMA);
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m instanceof Name && Objects.nonNull(comma)) {
                    EnumConstantDeclaration declare = new EnumConstantDeclaration(node.getPrarent());
                    declare.setName((Name) m);
                    declare.getChildrens().add(m);
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

                    if (a.getChildrens().isEmpty()) {
                        node.replace(a, declare);
                        c.clear();
                    } else {
                        int index = node.getChildrens().indexOf(a);
                        node.getChildrens().add(index, declare);
                        list.add(0, a);
                    }
                    comma = null;
                } else if (m.equals(TokenType.COMMA)) {
                    comma = m;
                    a.getChildrens().remove(m);
                }
            });
        });
    }

}