package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ELSE;
import static com.deep.framework.ast.lexer.TokenType.IF;

@Data
public class IfStatement extends Statement {
    private Expression condition;
    private List thenStatement;
    private Statement elseStatement;
    private Statement body;
    private static IfStatement statement;

    public static void parser(Node node) {
        if (node instanceof IfStatement) return;
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n) -> {
            if (m instanceof Node a) {
                if (Objects.isNull(statement) && a.getChildrens().contains(IF)) {
                    statement = new IfStatement();
                    statement.setPrarent(node);
                    statement.setChildrens(a.getChildrens());
                    statement.getChildrens().remove(IF);
                    if (Objects.nonNull(n) && n instanceof BlockStatement) {
                        statement.setBody((Statement) n);
                        node.getChildrens().remove(n);
                        statement.getChildrens().add(n);
                        ((Node) n).setPrarent(statement);
                        node.replaceAndRemove(a, statement, n);
                        list.remove(n);
                    } else {
                        node.replace(a, statement);
                    }
                } else if (a.getChildrens().contains(ELSE) && a.getChildrens().contains(IF)) {
                    a.setPrarent(statement);
                    statement.getChildrens().add(m);
                    statement.getChildrens().remove(List.of(ELSE, IF));
                    if (Objects.nonNull(n) && n instanceof BlockStatement) {
                        a.getChildrens().add(n);
                        ((Node) n).setPrarent(a);
                        node.getChildrens().removeAll(List.of(m, n));
                        list.remove(n);
                    } else {
                        node.getChildrens().remove(m);
                    }
                } else if (a.getChildrens().contains(ELSE)) {
                    a.setPrarent(statement);
                    statement.getChildrens().add(m);
                    statement.getChildrens().remove(ELSE);
                    if (Objects.nonNull(n) && n instanceof BlockStatement) {
                        a.getChildrens().add(n);
                        ((Node) n).setPrarent(a);
                        node.getChildrens().removeAll(List.of(m, n));
                        list.remove(n);
                    } else {
                        node.getChildrens().remove(m);
                    }
                    statement = null;
                }
            }
        });
    }
}