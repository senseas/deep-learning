package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ELSE;
import static com.deep.framework.ast.lexer.TokenType.IF;

@Data
public class IfStatement extends Statement {
    private ParametersExpression parameters;
    private NodeList thenStatement;
    private Statement elseStatement;
    private Statement body;
    private static IfStatement statement;

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(IF) && n instanceof ParametersExpression) {
                    statement = new IfStatement();
                    statement.setPrarent(node);
                    statement.setParameters((ParametersExpression) n);
                    node.replace(a, statement);

                    if (b instanceof BlockStatement) {
                        b.setPrarent(statement);
                        statement.setBody((BlockStatement) b);
                        statement.getChildrens().addAll(List.of(n, b));
                        node.getChildrens().remove(b);
                        list.remove(b);
                        c.clear();
                    } else {
                        BlockStatement block = new BlockStatement(statement);
                        block.setChildrens(a.getChildrens());
                        statement.setBody(block);
                        statement.getChildrens().addAll(List.of(n, block));
                        c.clear();
                    }

                } else if (m.equals(ELSE) && Objects.nonNull(n) && n.equals(IF)) {
                    a.setPrarent(statement);
                    a.getChildrens().removeAll(List.of(m, n));
                    statement.getChildrens().add(a);

                    if (b instanceof BlockStatement) {
                        a.getChildrens().add(b);
                        b.setPrarent(a);
                        node.getChildrens().removeAll(List.of(a, b));
                        list.remove(b);
                        c.clear();
                    } else {
                        node.getChildrens().remove(a);
                        c.clear();
                    }
                } else if (m.equals(ELSE)) {
                    a.setPrarent(statement);
                    a.getChildrens().remove(m);
                    statement.getChildrens().add(a);

                    if (b instanceof BlockStatement) {
                        a.getChildrens().add(b);
                        b.setPrarent(a);
                        node.getChildrens().removeAll(List.of(a, b));
                        list.remove(b);
                        c.clear();
                    } else {
                        node.getChildrens().remove(a);
                        c.clear();
                    }
                }
            });
        });
    }
}